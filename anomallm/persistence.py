import os
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import chainlit as cl
from chainlit.data import BaseDataLayer
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()

class DBUser(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)
    username = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class DBThread(Base):
    __tablename__ = "threads"
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(JSON)

class DBStep(Base):
    __tablename__ = "steps"
    id = Column(String, primary_key=True)
    thread_id = Column(String, ForeignKey("threads.id"))
    name = Column(String)
    type = Column(String)
    output = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    input = Column(Text)

class SQLiteDataLayer(BaseDataLayer):
    """
    Industrial-grade SQLite Data Layer for OmniML.
    Provides ChatGPT-like persistent chat history.
    """
    def __init__(self, db_path="sqlite:///database.db"):
        self.engine = create_engine(db_path, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    # User Management
    async def get_user(self, identifier: str):
        with self.Session() as s:
            user = s.query(DBUser).filter(DBUser.username == identifier).first()
            if user:
                return cl.User(id=user.id, identifier=user.username)
        return None

    async def create_user(self, user: cl.User):
        with self.Session() as s:
            new_user = DBUser(id=user.id or str(uuid.uuid4()), username=user.identifier)
            s.add(new_user)
            s.commit()
            return user

    # Thread Management
    async def create_thread(self, thread_id: str, user_id: str = None, metadata: Dict = None):
        with self.Session() as s:
            t = DBThread(id=thread_id, user_id=user_id, metadata_json=metadata)
            s.add(t)
            s.commit()
        return True

    async def get_thread(self, thread_id: str):
        with self.Session() as s:
            t = s.query(DBThread).get(thread_id)
            if t:
                # Chainlit 2.x strictly expects this dictionary structure to avoid React Native null bounds errors
                return {
                    "id": t.id, 
                    "createdAt": t.created_at.isoformat() + "Z" if t.created_at else datetime.utcnow().isoformat() + "Z",
                    "name": t.name or "New OmniML Run",
                    "userIdentifier": "Guest",
                    "metadata": t.metadata_json or {}, 
                    "steps": [],
                    "elements": []
                }
        return None

    async def list_threads(self, pagination, filter):
        with self.Session() as s:
            query = s.query(DBThread).order_by(DBThread.created_at.desc())
            threads = query.limit(20).all()
            result = []
            for t in threads:
                # Use standard dictionary to prevent framework attribute errors
                result.append({
                    "id": t.id,
                    "createdAt": t.created_at.isoformat() + "Z" if t.created_at else datetime.utcnow().isoformat() + "Z",
                    "name": t.name if t.name else "New OmniML Run",
                    "userIdentifier": "Guest",
                    "metadata": t.metadata_json if t.metadata_json else {}
                })
            return result

    async def update_thread(self, thread_id: str, name: str = None, user_id: str = None, metadata: Dict = None):
        with self.Session() as s:
            t = s.query(DBThread).get(thread_id)
            if t:
                if name: t.name = name
                if metadata: t.metadata_json = metadata
                s.commit()

    async def delete_thread(self, thread_id: str):
        with self.Session() as s:
            t = s.query(DBThread).get(thread_id)
            if t:
                # Also delete associated steps
                s.query(DBStep).filter(DBStep.thread_id == thread_id).delete()
                s.delete(t)
                s.commit()

    # Step Management
    async def create_step(self, step_dict: Dict):
        with self.Session() as s:
            name = step_dict.get("name")
            if name is None: name = "OmniML Step"
            
            step = DBStep(
                id=step_dict.get("id"),
                thread_id=step_dict.get("threadId"),
                name=name,
                type=step_dict.get("type") or "run",
                output=step_dict.get("output") or "",
                input=step_dict.get("input") or "",
                start_time=datetime.fromisoformat(step_dict.get("start")) if step_dict.get("start") else datetime.utcnow(),
                end_time=datetime.fromisoformat(step_dict.get("end")) if step_dict.get("end") else None
            )
            s.add(step)
            s.commit()

    async def update_step(self, step_dict: Dict):
         with self.Session() as s:
            step = s.query(DBStep).get(step_dict.get("id"))
            if step:
                if "output" in step_dict: 
                    step.output = step_dict.get("output") or ""
                if "name" in step_dict and step_dict.get("name"):
                    step.name = step_dict.get("name")
                if "end" in step_dict and step_dict.get("end"):
                    step.end_time = datetime.fromisoformat(step_dict.get("end"))
                s.commit()

    # Required abstract methods for Chainlit 2.x persistence
    async def delete_step(self, step_id: str):
        with self.Session() as s:
            s.query(DBStep).filter(DBStep.id == step_id).delete()
            s.commit()

    async def create_element(self, element: Any):
        # Prevent 'null is not an object' crash by ensuring elements have valid URLs
        # Copy to /public so Chainlit can serve it while custom persistence is active.
        el_url = element.get("url") if isinstance(element, dict) else getattr(element, "url", None)
        el_name = element.get("name") if isinstance(element, dict) else getattr(element, "name", "file")
        el_path = element.get("path") if isinstance(element, dict) else getattr(element, "path", None)

        if el_url is None:
            if el_path:
                try:
                    import shutil
                    os.makedirs("public", exist_ok=True)
                    public_loc = os.path.join("public", el_name)
                    shutil.copy(el_path, public_loc)
                    new_url = f"/public/{el_name}"
                    if isinstance(element, dict): element["url"] = new_url
                    else: element.url = new_url
                except Exception:
                    if isinstance(element, dict): element["url"] = ""
                    else: element.url = ""
            else:
                if isinstance(element, dict): element["url"] = ""
                else: element.url = ""
    async def get_element(self, thread_id: str, element_id: str): pass
    async def delete_element(self, thread_id: str, element_id: str): pass
    async def upsert_feedback(self, feedback: Any): pass
    async def delete_feedback(self, feedback_id: str): pass
    async def get_thread_author(self, thread_id: str): return "Guest_User"
    async def get_favorite_steps(self, user_id: str): return []
    async def set_step_favorite(self, step_id: str, favorite: bool): pass
    async def get_thread_author(self, thread_id: str): return "Guest_User"
    async def close(self): pass
    def build_debug_url(self) -> str: return ""

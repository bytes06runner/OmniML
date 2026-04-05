FROM python:3.12-slim

# Install system dependencies required for OpenCV or other data science libraries (if any pop up)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
# Hugging Face Spaces require running the container as a non-root user.
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# Install the Python dependencies from your existing robust requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary empty directories that OmniML saves logic to (so permissions are correct)
RUN mkdir -p $HOME/app/data && chown -R user:user $HOME/app/data

# Expose the default Hugging Face Space port
EXPOSE 7860

# Command to run on startup
# We run chainlit on port 7860 and listen on all interfaces
CMD ["chainlit", "run", "app.py", "--port", "7860", "--host", "0.0.0.0"]

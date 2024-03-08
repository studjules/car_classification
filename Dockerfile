
FROM python:3.11.6

ADD GUI_CarsCNN.py .

WORKDIR /app

# Copy the Pipfile and Pipfile.lock to the container
COPY . .

RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y libx11-6 libxext-dev libxrender-dev libxinerama-dev libxi-dev libxrandr-dev libexcursor-dev libxtst-dev tk-dev && rm -rf /var/lib/apt/lists/*

# Set the entry point for the application
CMD ["python", "GUI_CarsCNN.py"]

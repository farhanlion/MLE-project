# last updated Mar 25 2025, 11:00am
FROM python:3.12-slim

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Java (OpenJDK 17 headless), procps (for 'ps') and bash
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jdk-headless procps bash && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure Spark’s scripts run with bash instead of dash
    ln -sf /bin/bash /bin/sh && \
    # Create expected JAVA_HOME directory and symlink the java binary there
    mkdir -p /usr/lib/jvm/java-17-openjdk-amd64/bin && \
    ln -s "$(which java)" /usr/lib/jvm/java-17-openjdk-amd64/bin/java

# Set JAVA_HOME to the directory expected by Spark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install Python dependencies (ensure that pyspark is in your requirements.txt,
# or you can install it explicitly by uncommenting the next line)
RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install pyspark

# Expose the default JupyterLab port
EXPOSE 8888

# Create a volume mount point for notebooks
VOLUME /app

# Enable JupyterLab via environment variable
ENV JUPYTER_ENABLE_LAB=yes

# Set up the command to run JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]

# Use the AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.12

# Set the working directory to Lambda's default code directory
WORKDIR /var/task

# Copy only requirements first to leverage Docker cache
COPY requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY common ./common
COPY live ./live
COPY src ./src
# Copy any other modules your orchestrators depend on:
# COPY live ./live
# COPY training ./training
# If you need schema or static config:
COPY common/database ./common/database

# Set PYTHONPATH so `import common.db`, etc, works
ENV PYTHONPATH="/var/task"

# Default handler for the image. This is just a default.
# Each Lambda function can override it in the console.
CMD ["live.ingest.orchestrator.handler"]

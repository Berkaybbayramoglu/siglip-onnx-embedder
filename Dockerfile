FROM public.ecr.aws/lambda/python:3.12

# Copy requirements
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy function code and models (run `make export` before building)
COPY app/ ${LAMBDA_TASK_ROOT}/app/
COPY models/ ${LAMBDA_TASK_ROOT}/models/

CMD [ "app.handler.handler" ]

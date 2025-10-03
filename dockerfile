FROM python:3.13.7

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app
ADD . /app

# install uv
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

RUN uv sync --locked

# setup database
RUN alembic revision --autogenerate && alembic upgrade --head

# run
CMD ["uv", "run", "src/main.py"]

EXPOSE 8000
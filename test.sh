#!/bin/bash

docker kill test_db

docker run -d --rm -p 5432:5432 --env-file .env_db --name test_db postgres:16 \
&& sleep 2

alembic upgrade head \
&& pytest src/test/

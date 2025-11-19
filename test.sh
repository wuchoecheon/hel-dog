#!/bin/bash

docker kill test_db && docker rm test_db

docker run -d --restart always -p 5432:5432 --env-file .env.db --name test_db postgres:16 \
&& sleep 2

alembic upgrade head \
&& pytest src/test/
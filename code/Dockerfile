FROM python:3.8

RUN pip install pipenv
RUN mkdir usr/app/
RUN mkdir usr/app/data
RUN mkdir usr/app/static
RUN mkdir usr/app/static/images
COPY static/images usr/app/static/images
COPY data usr/app/data
COPY app.py usr/app/
COPY Pipfile usr/app/
COPY Pipfile.lock usr/app/
WORKDIR usr/app/
RUN pipenv install --system --deploy --ignore-pipfile

CMD ["pipenv", "run", "streamlit", "run", "app.py", "--server.port", "$PORT"]

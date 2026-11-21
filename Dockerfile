# Same base image (by digest) the commits were labeled in: its Python decides
# which sqlglot tests run and how.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# The exact package set the tests were labeled with (PyPI only).
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir --disable-pip-version-check -r /app/requirements.lock

# sqlglot at the pinned commit, from sqlglot's own GitHub repository.
ARG SQLGLOT_SHA=eb11aec4be8cb2122afe713a4816662c3f0fb2fa
RUN python -c "import io,os,shutil,tarfile,urllib.request as u; sha='${SQLGLOT_SHA}'; data=u.urlopen('https://github.com/tobymao/sqlglot/archive/'+sha+'.tar.gz', timeout=180).read(); tarfile.open(fileobj=io.BytesIO(data)).extractall('/opt', filter='data'); shutil.move('/opt/sqlglot-'+sha, '/opt/sqlglot')"

# Byte-compile the checkout and the selector so nothing compiles at run time.
COPY selector /app/selector
COPY maps /app/maps
RUN python -m compileall -q /opt/sqlglot/sqlglot /opt/sqlglot/tests /app/selector \
    && chmod -R a+rX /opt/sqlglot /app

# Sanity check at build time: the baseline index matches this checkout.
RUN cd /app && python -m selector.verify /opt/sqlglot

# Fingerprints of the pinned checkout's import-time state (class attributes,
# module globals), taken with this image's Python so they compare exactly with
# the patched state the selector fingerprints at run time.
RUN cd /app && python -m selector.statebuild /opt/sqlglot /app/maps/state.pkl \
    && chmod a+r /app/maps/state.pkl /app/maps/modules.json

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BR_REPO=/opt/sqlglot \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
WORKDIR /app
CMD ["python", "-m", "selector.main"]

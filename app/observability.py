import logfire


def setup_observability(app):
    logfire.configure(
        service_name="aws-case-studies-rag",
    )

    logfire.instrument_fastapi(app)
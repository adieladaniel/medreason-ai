"""
Alembic environment: which database to migrate and what the models should look like.

The URL comes from the application settings unless the caller passes one (`config.attributes["url"]`, which is how the service
and the tests migrate a specific database). The engine is the application's own (app/database/session.py), so a migration runs
with the same SQLite settings as the service. On SQLite, `render_as_batch` makes later ALTER TABLE migrations work: SQLite cannot
alter most things in place, so Alembic rebuilds the table instead. Logging is left to the application (observability.py): calling
logging.config.fileConfig here would replace its handlers.
"""
from alembic import context

from app.config import get_settings
from app.database.models import Base
from app.database.session import make_engine

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    return config.attributes.get("url") or get_settings().database_url


def run_migrations_offline() -> None:
    """Write the SQL instead of running it (`alembic upgrade head --sql`)."""
    url = _url()
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True, dialect_opts={"paramstyle": "named"},
                      render_as_batch=url.startswith("sqlite"))
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    url = _url()
    with make_engine(url).connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True,
                          render_as_batch=url.startswith("sqlite"))
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

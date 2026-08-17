from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

from app.config.settings import settings
from app.db.base import Base
import app.models

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(
        config.config_ini_section,
        {}
    )

    configuration["sqlalchemy.url"] = settings.database_url

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()


"""
rohit notes:
(1) import app.models
This imports all models from app/models/__init__.py, so all tables get registered in: Base.metadata

(2)
Find this line
target_metadata = None

Replace it with:

target_metadata = Base.metadata

(3)
Find run_migrations_online()

Replace the entire function with:

def run_migrations_online() -> None:
    configuration = config.get_section(
        config.config_ini_section,
        {}
    )


    configuration["sqlalchemy.url"] = settings.database_url


    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )


    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )


        with context.begin_transaction():
            context.run_migrations()
Your flow is now
.env
  ↓
DATABASE_URL
  ↓
settings.database_url
  ↓
Alembic env.py
  ↓
PostgreSQL

And:

app.models
    ↓ imports
Document
DocumentVersion
Category
Tag
    ↓
Base.metadata
    ↓
Alembic detects tables

That includes the six tables:

documents
document_versions
categories
document_categories
tags
document_tags


(4)
After updating env.py, run:

uv run alembic revision --autogenerate -m "create document tables"

This should generate a migration inside:

alembic/versions/


(5)
Next step: apply the migration

Run:

uv run alembic upgrade head

This will execute the upgrade() function and create the tables in your contextops database:

contextops
├── alembic_version
├── categories
├── documents
├── tags
├── document_categories
├── document_tags
└── document_versions

alembic_version is created by Alembic to track which migration has been applied.
"""
import sys
sys.path.append("../")

# from threading import Event, Thread
from sqlalchemy import create_engine, MetaData
# , text
from sqlalchemy.dialects.postgresql import ENUM

from utils.config_helper import rds_conn, redshift_conn


# we have separate engines for rds and redshift
# so a lot of requests in one queue won't affect the other queue.
def rds_init():
    engine = create_engine('postgresql://', creator=rds_conn,
                           pool_size=10, pool_pre_ping=True)
    m = MetaData(engine, schema='public')
    m.reflect(engine)
    status_enum = ENUM('pending', 'completed', 'error', name='status',
                       metadata=m)
    status_enum.create(engine)
    return engine

def redshift_init():
        return create_engine(
                        'postgresql://', creator=redshift_conn, pool_size=30, pool_pre_ping=True)


#rds_engine = rds_init()
redshift_engine = redshift_init()
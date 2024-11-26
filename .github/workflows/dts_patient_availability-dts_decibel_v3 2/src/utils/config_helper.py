import base64
import logging
import logging.config
import yaml
import os
import json
import boto3
import psycopg2

logger = logging.getLogger(__name__)
redshift_user_name = None


def getRegion():
    if 'AWS_DEFAULT_REGION' not in os.environ:
        return 'us-east-1'
    else:
        return os.environ['AWS_DEFAULT_REGION']


def get_secret_value():
    secrets_client = boto3.client('secretsmanager', region_name=getRegion())
    secretId = os.environ['SECRET_ID']
    return secrets_client.get_secret_value(SecretId=secretId)


def redshift_conn():
    return psycopg2.connect(**getRedshiftConnectionString(get_dict=True))


def rds_conn():
    return psycopg2.connect(**rds_connection_str(get_dict=True))


def getRedshiftConnectionString(get_dict=False):
    try:
        secretValue = get_secret_value()
        info = json.loads(secretValue['SecretString'])['analytics_interface'][
            'database']
        logger.info(f'redshift connection string found in secret manager')
        logger.info(f"connecting to {info['dbname']} : {info['host']}")
    except Exception:
        logger.warning("redshift connection string not found in secret manager")
        info = {
            'dbname': os.environ.get('REDSHIFT_DBNAME', 'concerto_dev'),
            'host': os.environ.get('REDSHIFT_HOST',
                                   'concerto.cfaw7z5r1wcw.us-east-1.redshift.amazonaws.com'),
            'port': '5439',
            'user': os.environ['REDSHIFT_USER'],
            'password': os.environ['REDSHIFT_PASSWORD']
        }
    logger.info(f"redshift host: {info['host']}")
    logger.info(f"redshift dbname: {info['dbname']}")
    logger.info(f"redshift user: {info['user']}")
    logger.info(f'eureka_be app has num_cpus = {os.cpu_count()}')
    global redshift_user_name
    redshift_user_name = str(info['user']).strip()
    return info if get_dict else 'redshift://{user}:{password}@{host}:{port}/{dbname}'.format(**info)


def get_redshift_user_name():
    return redshift_user_name

def rds_connection_str_celery(get_dict=False):
    try:
        import urllib
        logger.info("Searching for RDS connection string")
        secretValue = get_secret_value()
        info = json.loads(secretValue['SecretString'])['application_database'][
            'database']
        logger.info(f'RDS connection string found in secret manager')
        logger.info(f"connecting to {info['dbname']} : {info['host']}")
    except Exception:
        logger.warning("RDS connection string not found in secret manager")
        info = {
            'dbname': os.environ.get('APPLICATION_DB_NAME', 'concertostage1'),
            'host': os.environ.get('APPLICATION_DB_HOST',
                                   'concerto.c5rll1tuidz0.us-east-1.rds.amazonaws.com'),
            'port': '5432',
            'user': os.environ['APPLICATION_DB_USER'],
            'password': os.environ['APPLICATION_DB_PASSWORD']
        }
    info['password'] = urllib.parse.quote_plus(info['password'])
    logger.info(f"application db host: {info['host']}")
    logger.info(f"application db dbname: {info['dbname']}")
    logger.info(f"application db user: {info['user']}")
    return info if get_dict else 'postgres://{user}:{password}@{host}:{port}/{dbname}'.format(**info)

def rds_connection_str(get_dict=False):
    try:
        logger.info("Searching for RDS connection string")
        secretValue = get_secret_value()
        info = json.loads(secretValue['SecretString'])['application_database'][
            'database']
        logger.info(f'RDS connection string found in secret manager')
        logger.info(f"connecting to {info['dbname']} : {info['host']}")
    except Exception:
        logger.warning("RDS connection string not found in secret manager")
        info = {
            'dbname': os.environ.get('APPLICATION_DB_NAME', 'concertostage1'),
            'host': os.environ.get('APPLICATION_DB_HOST',
                                   'concerto.c5rll1tuidz0.us-east-1.rds.amazonaws.com'),
            'port': '5432',
            'user': os.environ['APPLICATION_DB_USER'],
            'password': os.environ['APPLICATION_DB_PASSWORD']
        }
    logger.info(f"application db host: {info['host']}")
    logger.info(f"application db dbname: {info['dbname']}")
    logger.info(f"application db user: {info['user']}")
    return info if get_dict else 'postgres://{user}:{password}@{host}:{port}/{dbname}'.format(**info)


def setup_logger(config_path, default_level=logging.INFO):
    from importlib import reload
    reload(logging)
    reload(logging.config)
    if os.path.exists(config_path):
        with open(config_path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    # if file not found, use basic config to avoid exceptions
    else:
        print("Using default config")
        logging.basicConfig(level=default_level)
    # Remove the below line if we want info logs from sqlalchemy.
    logging.getLogger('sqlalchemy').setLevel(logging.ERROR)
    return logging


def get_table_secret_key(table_name):
    def get_key_for_table(info, table_name):
        return info['rwe_keys'][table_name]
    try:
        logger.info("Searching for AES key")
        secretValue = get_secret_value()
        info = get_key_for_table(json.loads(secretValue['SecretString']), table_name)
        logger.info(f'AES key found in secret manager')
    except Exception:
        logger.warning("AES key for table " + table_name + " not found in secret manager")
        info = os.environ['PERSON_TABLE_KEY']
    return info['key']


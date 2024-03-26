import sqlalchemy
from loguru import logger

from utils.config_util import get_starrocks_config


class StarrocksDbUtil:
    def __init__(self):
        self.engine = None

    def get_db_engine(self):
        if self.engine is None:
            self.engine = self._create_db_engine()
        return self.engine

    def _create_db_engine(self):
        server_address, port, db_name, user, password = get_starrocks_config()
        return sqlalchemy.create_engine(f"starrocks://{user}:{password}@{server_address}:{port}/{db_name}?charset=utf8", connect_args={'connect_timeout': 600})


    def table_exists(self, table_name):
        return sqlalchemy.inspect(self.get_db_engine()).has_table(table_name)


    def run_sql(self, sql, log=True, chunksize=None):
        if log:
            logger.info("run sql: {}".format(sql))
        sql = sqlalchemy.text(sql)
        result = []
        with self.get_db_engine().connect() as con:
            try:
                if chunksize:
                    # cursor_result = con.execution_options(yield_per=1).execute(sql)
                    # for partition in cursor_result.partitions():
                    #     for row in partition:
                    #         result.append(row)
                    cursor_result = con.execution_options(stream_results=True, max_row_buffer=100).execute(sql)
                    while True:
                        data = cursor_result.fetchmany(chunksize)
                        if not data:
                            break
                        result.extend(list(data))
                else:
                    cursor_result = con.execute(sql)
                    result = list(cursor_result)
            except Exception as e:
                # if there are no results returned, exception will be ignored.
                logger.info(e)

        return result



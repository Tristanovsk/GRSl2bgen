"""Log file of a GRSl2bgen run, in the format of the GRS processor (``grs.class_logger``).

Records look like::

    2026-10-20T09:20:46.123     INFO | root::write_output | export into encoded netcdf

and the run ends with its resource statistics (peak memory, CPU and run times) and the
number of records per level.
"""

import logging
import resource
import time

LOG_FORMAT = '%(asctime)s.%(msecs)03d     %(levelname)s | %(name)s::%(funcName)s | %(message)s'
DATE_FORMAT = '%Y-%m-%dT%H:%M:%S'


class _CountHandler(logging.Handler):
    """Count the records of each level."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.counts = {'ERROR': 0, 'WARNING': 0, 'INFO': 0, 'DEBUG': 0}

    def emit(self, record):
        if record.levelname in self.counts:
            self.counts[record.levelname] += 1


class RunLogger:
    """Send the root logger records to `log_file` (and the console) for one run.

    Parameters
    ----------
    log_file : str
        Log file path, overwritten.
    log_level : str, optional
        Lowest level written (default ``'INFO'``).
    log_console : bool, optional
        Also write the records to the console (stderr, default True).

    Notes
    -----
    Python warnings are logged as ``WARNING`` records. `close` logs the resource
    statistics of the process, removes the handlers and restores the handlers and level
    of the root logger found at the start (e.g. those of a notebook); the class is also a context
    manager, whose exit logs an uncaught exception before closing.
    """

    def __init__(self, log_file, log_level='INFO', log_console=True):
        self.log_file = log_file
        self.t0 = time.perf_counter()
        self.usage0 = self._usage()
        self.root_logger = logging.getLogger()
        self.previous = (self.root_logger.level, list(self.root_logger.handlers))
        self.root_logger.setLevel(log_level)
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

        # handlers of an earlier configuration (e.g. a notebook) would duplicate the
        # records: removed for the run, restored by `close`
        for handler in self.previous[1]:
            self.root_logger.removeHandler(handler)

        self.counter = _CountHandler()
        self.handlers = [self.counter, logging.FileHandler(log_file, mode='w')]
        if log_console:
            self.handlers.append(logging.StreamHandler())
        for handler in self.handlers[1:]:
            handler.setFormatter(formatter)
            handler.setLevel(log_level)
        for handler in self.handlers:
            self.root_logger.addHandler(handler)

        logging.captureWarnings(True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None and not issubclass(exc_type, (SystemExit, KeyboardInterrupt)):
            logging.getLogger().error('processing failed', exc_info=(exc_type, exc, tb))
        self.close()
        return False

    @staticmethod
    def _usage():
        """(peak RSS in kB, sys CPU in s, user CPU in s) of the process and its children."""
        usage = [resource.getrusage(who) for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
        return (max(u.ru_maxrss for u in usage), sum(u.ru_stime for u in usage),
                sum(u.ru_utime for u in usage))

    def log_stats(self):
        """Log the peak memory of the process and the CPU and run times of the run.

        The CPU times count from the creation of the logger, so that several runs in
        one process (e.g. a notebook batch) each get their own. The peak memory cannot
        be reset: it is the peak of the process since its start.
        """
        # record counts of the run itself, before the statistics records
        counts = dict(self.counter.counts)
        logger = logging.getLogger('service_logger')
        max_rss, sys_cpu, user_cpu = self._usage()
        logger.info(f'max_rss: {max_rss} kB')
        logger.info(f'sys_cpu: {sys_cpu - self.usage0[1]:.2f} s')
        logger.info(f'user_cpu: {user_cpu - self.usage0[2]:.2f} s')
        logger.info(f'total_run_time: {time.perf_counter() - self.t0:.2f} s')
        for level, count in counts.items():
            logger.info(f'{level.lower()}: {count}')

    def close(self):
        """Log the statistics, remove the handlers and restore the previous configuration."""
        self.log_stats()
        logging.captureWarnings(False)
        for handler in self.handlers:
            self.root_logger.removeHandler(handler)
            handler.close()
        level, handlers = self.previous
        self.root_logger.setLevel(level)
        for handler in handlers:
            self.root_logger.addHandler(handler)

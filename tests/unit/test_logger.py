"""Log file of a run (GRSl2bgen.logger.RunLogger)."""

import logging
import warnings

import pytest

from GRSl2bgen.logger import RunLogger


def test_run_logger_writes_records_and_stats(tmp_path):
    log_file = tmp_path / 'l2b.log'
    with RunLogger(log_file, log_console=False):
        logging.info('step one')
        warnings.warn('a warning')
    text = log_file.read_text()
    assert 'INFO | root::test_run_logger_writes_records_and_stats | step one' in text
    assert 'WARNING' in text and 'a warning' in text
    for key in ('max_rss:', 'user_cpu:', 'total_run_time:', 'info: 1', 'warning: 1', 'error: 0'):
        assert key in text, key
    # handlers removed after the run
    logging.info('after the run')
    assert 'after the run' not in log_file.read_text()


def test_run_logger_logs_uncaught_exception(tmp_path):
    log_file = tmp_path / 'l2b.log'
    with pytest.raises(RuntimeError):
        with RunLogger(log_file, log_console=False):
            raise RuntimeError('boom')
    text = log_file.read_text()
    assert 'ERROR' in text and 'processing failed' in text and 'RuntimeError: boom' in text
    assert 'error: 1' in text


def test_run_logger_restores_previous_handlers(tmp_path):
    root = logging.getLogger()
    previous = logging.StreamHandler()
    root.addHandler(previous)
    level = root.level
    try:
        with RunLogger(tmp_path / 'l2b.log', log_level='DEBUG', log_console=False):
            assert previous not in root.handlers
        assert previous in root.handlers and root.level == level
    finally:
        root.removeHandler(previous)

"""Tests for mullet.harness.observability.log_aggregator."""

import logging
import time


from mullet.harness.observability.log_aggregator import (
    LogAggregator,
    LogHandler,
    record,
    query,
    clear,
)


class TestLogAggregator:
    def test_record_and_query(self):
        agg = LogAggregator()
        agg.record(level="INFO", message="hello", source="test")
        agg.record(level="ERROR", message="boom", source="test")
        results = agg.query(level="ERROR")
        assert len(results) == 1
        assert results[0]["message"] == "boom"

    def test_query_by_source(self):
        agg = LogAggregator()
        agg.record(level="INFO", message="a", source="mod.a")
        agg.record(level="INFO", message="b", source="mod.b")
        assert len(agg.query(source="mod.a")) == 1
        assert agg.query(source="mod.a")[0]["message"] == "a"

    def test_query_since(self):
        agg = LogAggregator()
        before = time.time()
        agg.record(level="INFO", message="old", source="test")
        time.sleep(0.01)
        after = time.time()
        agg.record(level="INFO", message="new", source="test")
        results = agg.query(since=after)
        assert len(results) == 1
        assert results[0]["message"] == "new"

    def test_query_limit(self):
        agg = LogAggregator()
        for i in range(5):
            agg.record(level="INFO", message=str(i), source="test")
        results = agg.query(limit=2)
        assert len(results) == 2
        assert results[0]["message"] == "3"
        assert results[1]["message"] == "4"

    def test_max_records_eviction(self):
        agg = LogAggregator(max_records=3)
        for i in range(5):
            agg.record(level="INFO", message=str(i), source="test")
        assert len(agg.records) == 3
        assert agg.records[0]["message"] == "2"
        assert agg.records[-1]["message"] == "4"

    def test_clear(self):
        agg = LogAggregator()
        agg.record(level="INFO", message="x", source="test")
        removed = agg.clear()
        assert removed == 1
        assert agg.records == []

    def test_extras(self):
        agg = LogAggregator()
        agg.record(level="INFO", message="m", source="s", extra_key=42)
        assert agg.records[0]["extra_key"] == 42


class TestLogHandler:
    def test_emit_forwards_to_aggregator(self):
        agg = LogAggregator()
        handler = LogHandler(aggregator=agg)
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger("test_emit_logger")
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        logger.addHandler(handler)
        logger.info("hello handler")
        results = agg.query(source="test_emit_logger")
        assert len(results) == 1
        assert "hello handler" in results[0]["message"]


class TestModuleLevelSingleton:
    def test_round_trip(self):
        clear()
        record(level="WARN", message="singleton", source="mod")
        results = query(level="WARN")
        assert len(results) == 1
        assert results[0]["message"] == "singleton"
        clear()

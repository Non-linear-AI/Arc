"""Tests for DatabaseManager."""

import threading
import time
from pathlib import Path

import pytest

from arc.database import DatabaseError, DatabaseManager


def test_database_manager_initialization():
    """Test DatabaseManager initialization."""
    system_db = ":memory:"
    user_db = ":memory:"

    with DatabaseManager(system_db, user_db) as manager:
        assert manager.get_system_db_path() == system_db
        assert manager.get_user_db_path() == user_db
        assert manager.has_user_database()


def test_database_manager_no_user_db():
    """Test DatabaseManager with no user database."""
    system_db = ":memory:"

    with DatabaseManager(system_db) as manager:
        assert manager.get_system_db_path() == system_db
        assert manager.get_user_db_path() is None
        assert not manager.has_user_database()


def test_system_database_operations(tmp_path):
    """Test system database operations."""
    system_db = tmp_path / "system.db"
    with DatabaseManager(str(system_db)) as manager:
        # System database should have Arc schema initialized
        result = manager.system_query("SELECT COUNT(*) as count FROM models")
        assert result.first()["count"] == 0

        # Insert a test model
        manager.system_execute("""
            INSERT INTO models (id, name, version, type)
            VALUES ('test-1', 'test_model', 1, 'classification')
        """)

        result = manager.system_query("SELECT * FROM models WHERE id = 'test-1'")
        assert result.count() == 1
        assert result.first()["name"] == "test_model"


def test_user_database_operations(tmp_path):
    """Test user database operations."""
    system_db = tmp_path / "system.db"
    user_db = tmp_path / "user.db"
    with DatabaseManager(str(system_db), str(user_db)) as manager:
        # Create a test table in user database
        manager.user_execute("CREATE TABLE test_data (id INTEGER, value TEXT)")

        # Insert test data
        manager.user_execute("INSERT INTO test_data VALUES (1, 'test')")

        result = manager.user_query("SELECT * FROM test_data")
        assert result.count() == 1
        assert result.first()["value"] == "test"


def test_user_database_not_configured(tmp_path):
    """Test operations when user database is not configured."""
    system_db = tmp_path / "system.db"
    with DatabaseManager(str(system_db)) as manager:
        with pytest.raises(DatabaseError, match="No user database configured"):
            manager.user_query("SELECT 1")

        with pytest.raises(DatabaseError, match="No user database configured"):
            manager.user_execute("CREATE TABLE test (id INTEGER)")


def test_set_user_database(tmp_path):
    """Test switching user databases."""
    system_db = tmp_path / "system.db"
    with DatabaseManager(str(system_db)) as manager:
        assert not manager.has_user_database()

        # Set user database
        user_db = tmp_path / "user.db"
        manager.set_user_database(str(user_db))
        assert manager.has_user_database()
        assert manager.get_user_db_path() == str(user_db)

        # Should now work
        manager.user_execute("CREATE TABLE test (id INTEGER)")
        result = manager.user_query("SELECT COUNT(*) as count FROM test")
        assert result.first()["count"] == 0


def test_pathlib_paths():
    """Test that Path objects are handled correctly."""
    system_path = Path(":memory:")
    user_path = Path(":memory:")

    with DatabaseManager(system_path, user_path) as manager:
        assert manager.get_system_db_path() == str(system_path)
        assert manager.get_user_db_path() == str(user_path)


def test_thread_local_connections():
    """Test that each thread gets its own database connections."""
    manager = DatabaseManager(":memory:", ":memory:")
    results = {}

    def worker_thread(thread_id):
        """Worker function that operates on the database."""
        # Create a test table and insert thread-specific data
        manager.user_execute(
            f"CREATE TABLE IF NOT EXISTS thread_test_{thread_id} (value TEXT)"
        )
        manager.user_execute(
            f"INSERT INTO thread_test_{thread_id} VALUES ('thread_{thread_id}')"
        )

        # Each thread should see its own data
        result = manager.user_query(f"SELECT value FROM thread_test_{thread_id}")
        results[thread_id] = result.first()["value"]

    # Create multiple threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker_thread, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify each thread got its own data
    for i in range(5):
        assert results[i] == f"thread_{i}"

    manager.close()


def test_concurrent_system_database_access(tmp_path):
    """Test concurrent access to system database from multiple threads."""
    system_db = tmp_path / "system.db"
    manager = DatabaseManager(str(system_db))
    results = []
    errors = []

    def system_worker(thread_id):
        """Worker function that performs system database operations."""
        try:
            # Insert a model record
            manager.system_execute(
                """
                INSERT INTO models (id, name, version, type)
                VALUES (?, ?, ?, ?)
            """,
                [f"model_{thread_id}", f"model_{thread_id}", 1, "classification"],
            )

            # Query the model back
            result = manager.system_query(
                "SELECT name FROM models WHERE id = ?", [f"model_{thread_id}"]
            )

            results.append((thread_id, result.first()["name"]))
        except Exception as e:
            errors.append((thread_id, str(e)))

    # Create multiple threads
    threads = []
    for i in range(10):
        thread = threading.Thread(target=system_worker, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Should have no errors
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Should have results from all threads
    assert len(results) == 10

    # Verify each thread's data
    for thread_id, name in results:
        assert name == f"model_{thread_id}"

    manager.close()


def test_mixed_concurrent_operations(tmp_path):
    """Test mixing system and user database operations across threads."""
    system_db = tmp_path / "system.db"
    user_db = tmp_path / "user.db"
    manager = DatabaseManager(str(system_db), str(user_db))
    system_results = []
    user_results = []
    errors = []

    def mixed_worker(thread_id):
        """Worker that performs both system and user operations."""
        try:
            # System operation - insert a job
            manager.system_execute(
                """
                INSERT INTO jobs (job_id, type, status, message)
                VALUES (?, ?, ?, ?)
            """,
                [f"job_{thread_id}", "test", "pending", f"Test job {thread_id}"],
            )

            # User operation - create table and insert data
            table_name = f"user_data_{thread_id}"
            manager.user_execute(
                f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER, name TEXT)"
            )
            manager.user_execute(
                f"INSERT INTO user_data_{thread_id} VALUES (?, ?)",
                [thread_id, f"user_{thread_id}"],
            )

            # Query both databases
            job_result = manager.system_query(
                "SELECT message FROM jobs WHERE job_id = ?", [f"job_{thread_id}"]
            )
            user_result = manager.user_query(
                f"SELECT name FROM user_data_{thread_id} WHERE id = ?", [thread_id]
            )

            system_results.append((thread_id, job_result.first()["message"]))
            user_results.append((thread_id, user_result.first()["name"]))

        except Exception as e:
            errors.append((thread_id, str(e)))

    # Create multiple threads
    threads = []
    for i in range(8):
        thread = threading.Thread(target=mixed_worker, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Should have no errors
    assert len(errors) == 0, f"Errors occurred: {errors}"

    # Should have results from all threads
    assert len(system_results) == 8
    assert len(user_results) == 8

    # Verify data integrity
    for thread_id, message in system_results:
        assert message == f"Test job {thread_id}"

    for thread_id, name in user_results:
        assert name == f"user_{thread_id}"

    manager.close()


def test_thread_isolation_with_database_switching(tmp_path):
    """Test that database switching is properly isolated per thread."""
    system_db = tmp_path / "system.db"
    manager = DatabaseManager(str(system_db))
    results = {}

    def worker_with_db_switch(thread_id):
        """Worker that switches user database and performs operations."""
        # Each thread sets its own "user database" (in memory)
        manager.set_user_database(":memory:")

        # Create thread-specific data
        manager.user_execute("CREATE TABLE thread_data (thread_id INTEGER, data TEXT)")
        manager.user_execute(
            "INSERT INTO thread_data VALUES (?, ?)", [thread_id, f"data_{thread_id}"]
        )

        # Brief delay to increase chance of thread interleaving
        time.sleep(0.01)

        # Verify the data is still there (not overwritten by other threads)
        result = manager.user_query(
            "SELECT data FROM thread_data WHERE thread_id = ?", [thread_id]
        )
        results[thread_id] = result.first()["data"]

    # Create multiple threads
    threads = []
    for i in range(6):
        thread = threading.Thread(target=worker_with_db_switch, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Each thread should have its own isolated data
    for i in range(6):
        assert results[i] == f"data_{i}"

    manager.close()


def test_connection_cleanup_per_thread(tmp_path):
    """Test that connections are properly cleaned up per thread."""
    system_db = tmp_path / "system.db"
    user_db = tmp_path / "user.db"
    manager = DatabaseManager(str(system_db), str(user_db))

    def worker_with_cleanup():
        """Worker that creates connections and then cleans them up."""
        # Access both databases to create connections
        manager.system_query("SELECT COUNT(*) FROM models")
        manager.user_execute("CREATE TABLE IF NOT EXISTS cleanup_test (id INTEGER)")

        # Explicitly close connections for this thread
        manager.close()

        # Should be able to access again (new connections will be created)
        manager.system_query("SELECT COUNT(*) FROM jobs")

    # Create and run thread
    thread = threading.Thread(target=worker_with_cleanup)
    thread.start()
    thread.join()

    # Main thread should still be able to access databases
    result = manager.system_query("SELECT COUNT(*) as count FROM models")
    assert result.first()["count"] == 0

    manager.close()

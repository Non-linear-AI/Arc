"""Basic tests for SchemaService."""

import pytest

from arc.database import DatabaseManager
from arc.database.base import DatabaseError
from arc.database.services.schema_service import (
    ColumnInfo,
    SchemaInfo,
    SchemaService,
    TableInfo,
)


@pytest.fixture
def db_manager():
    """Create an in-memory database manager for testing."""
    with DatabaseManager(":memory:", ":memory:") as manager:
        # Create some test tables in user database
        manager.user_execute("""
            CREATE TABLE test_users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        manager.user_execute("""
            CREATE TABLE test_orders (
                order_id INTEGER PRIMARY KEY,
                user_id INTEGER,
                amount DECIMAL(10,2),
                status TEXT DEFAULT 'pending'
            )
        """)
        yield manager


@pytest.fixture
def schema_service(db_manager):
    """Create a SchemaService instance for testing."""
    return SchemaService(db_manager)


class TestSchemaDataClasses:
    """Test the schema data classes."""

    def test_table_info_creation(self):
        """Test creating a TableInfo instance."""
        table = TableInfo(name="users", schema="main", table_type="BASE TABLE")
        assert table.name == "users"
        assert table.schema == "main"
        assert table.table_type == "BASE TABLE"

    def test_column_info_creation(self):
        """Test creating a ColumnInfo instance."""
        column = ColumnInfo(
            table_name="users",
            column_name="id",
            data_type="INTEGER",
            is_nullable=False,
            default_value=None,
            column_position=1
        )
        assert column.table_name == "users"
        assert column.column_name == "id"
        assert column.data_type == "INTEGER"
        assert not column.is_nullable

    def test_schema_info_methods(self):
        """Test SchemaInfo utility methods."""
        tables = [
            TableInfo(name="users"),
            TableInfo(name="orders")
        ]
        columns = [
            ColumnInfo(table_name="users", column_name="id", data_type="INTEGER"),
            ColumnInfo(table_name="users", column_name="name", data_type="TEXT"),
            ColumnInfo(table_name="orders", column_name="id", data_type="INTEGER")
        ]
        
        schema = SchemaInfo(tables=tables, columns=columns)
        
        # Test table methods
        assert schema.get_table_names() == ["users", "orders"]
        assert schema.table_exists("users")
        assert not schema.table_exists("nonexistent")
        
        # Test column methods
        user_columns = schema.get_columns_for_table("users")
        assert len(user_columns) == 2
        assert schema.get_column_names("users") == ["id", "name"]


class TestSchemaService:
    """Test SchemaService basic functionality."""

    def test_schema_service_initialization(self, db_manager):
        """Test SchemaService initialization."""
        service = SchemaService(db_manager)
        assert service.db_manager == db_manager
        assert service._system_schema_cache is None
        assert service._user_schema_cache is None

    def test_get_schema_info_system(self, schema_service):
        """Test getting system database schema."""
        schema_info = schema_service.get_schema_info("system")
        
        assert isinstance(schema_info, SchemaInfo)
        assert len(schema_info.tables) > 0
        
        # System should have Arc's tables (models, jobs, etc.)
        table_names = schema_info.get_table_names()
        assert "models" in table_names
        assert "jobs" in table_names

    def test_get_schema_info_user(self, schema_service):
        """Test getting user database schema."""
        schema_info = schema_service.get_schema_info("user")
        
        assert isinstance(schema_info, SchemaInfo)
        table_names = schema_info.get_table_names()
        
        # Should find our test tables
        assert "test_users" in table_names
        assert "test_orders" in table_names

    def test_get_schema_info_invalid_db(self, schema_service):
        """Test getting schema for invalid database."""
        with pytest.raises(DatabaseError, match="Invalid target database"):
            schema_service.get_schema_info("invalid")

    def test_schema_caching(self, schema_service):
        """Test schema caching functionality."""
        # First call should populate cache
        schema1 = schema_service.get_schema_info("user")
        assert schema_service._user_schema_cache is not None
        
        # Second call should use cache
        schema2 = schema_service.get_schema_info("user")
        assert schema1 is schema2  # Same object from cache
        
        # Force refresh should update cache
        schema3 = schema_service.get_schema_info("user", force_refresh=True)
        assert schema3 is not schema2  # Different object

    def test_get_table_summary_all_tables(self, schema_service):
        """Test getting summary of all tables."""
        summary = schema_service.get_table_summary("user")
        
        assert "User Database Schema:" in summary
        assert "test_users" in summary
        assert "test_orders" in summary
        assert "columns" in summary

    def test_get_table_summary_specific_table(self, schema_service):
        """Test getting summary of specific table."""
        summary = schema_service.get_table_summary("user", "test_users")
        
        assert "Table: test_users" in summary
        assert "id: INTEGER" in summary
        assert "name: VARCHAR" in summary
        assert "NOT NULL" in summary

    def test_get_table_summary_nonexistent_table(self, schema_service):
        """Test getting summary of nonexistent table."""
        summary = schema_service.get_table_summary("user", "nonexistent")
        assert "not found" in summary

    def test_is_ddl_statement(self, schema_service):
        """Test DDL statement detection."""
        # DDL statements
        assert schema_service.is_ddl_statement("CREATE TABLE test (id INT)")
        assert schema_service.is_ddl_statement("DROP TABLE test")
        assert schema_service.is_ddl_statement("ALTER TABLE test ADD COLUMN name TEXT")
        assert schema_service.is_ddl_statement("TRUNCATE TABLE test")
        
        # Non-DDL statements
        assert not schema_service.is_ddl_statement("SELECT * FROM test")
        assert not schema_service.is_ddl_statement("INSERT INTO test VALUES (1)")
        assert not schema_service.is_ddl_statement("UPDATE test SET name = 'test'")
        assert not schema_service.is_ddl_statement("")

    def test_cache_invalidation(self, schema_service):
        """Test cache invalidation methods."""
        # Populate caches
        schema_service.get_schema_info("system")
        schema_service.get_schema_info("user")
        
        assert schema_service._system_schema_cache is not None
        assert schema_service._user_schema_cache is not None
        
        # Test specific cache invalidation
        schema_service.invalidate_cache("user")
        assert schema_service._system_schema_cache is not None
        assert schema_service._user_schema_cache is None
        
        # Test clear all caches
        schema_service.get_schema_info("user")  # Repopulate
        schema_service.clear_cache()
        assert schema_service._system_schema_cache is None
        assert schema_service._user_schema_cache is None

    def test_generate_system_schema_prompt(self, schema_service):
        """Test generating system schema prompt."""
        prompt = schema_service.generate_system_schema_prompt()
        
        assert "System Database Schema" in prompt
        assert "models" in prompt
        assert "jobs" in prompt
        assert "Usage Guidelines" in prompt

    def test_validate_query_schema_basic(self, schema_service):
        """Test basic query validation."""
        result = schema_service.validate_query_schema(
            "SELECT * FROM test_users", 
            "user"
        )
        
        assert "valid" in result
        assert "referenced_tables" in result
        assert "available_tables" in result
        assert "test_users" in result["available_tables"]

package miniproject;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Tests for database persistence and transaction management.
 */
public class DatabaseTest {

    @Test
    public void testTransactionRollback() {
        // Ensures that a failed transaction rolls back all changes
        boolean rolledBack = Database.performAndRollback();
        assertTrue("Transaction should have been rolled back", rolledBack);
        int rowCount = Database.countRows();
        assertEquals("No rows should remain after rollback", 0, rowCount);
    }
}

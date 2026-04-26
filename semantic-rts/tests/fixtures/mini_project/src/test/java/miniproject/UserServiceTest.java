package miniproject;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Tests for the UserService business logic layer.
 */
public class UserServiceTest {

    @Test
    public void testCreateUser() {
        User user = UserService.create("alice", "alice@example.com");
        assertNotNull("Created user should not be null", user);
        assertEquals("alice", user.getName());
        assertEquals("alice@example.com", user.getEmail());
    }
}

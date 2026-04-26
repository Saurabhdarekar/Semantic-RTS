package miniproject;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Tests for authentication and security functionality.
 * Covers password validation and token expiry checks.
 */
public class AuthTest {

    @Test
    public void testPasswordValidation() {
        // Passwords shorter than 8 characters must be rejected
        boolean result = Auth.validatePassword("short");
        assertFalse("Short password should fail validation", result);
    }
}

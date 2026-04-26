package miniproject;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Tests for string formatting and utility helpers.
 */
public class StringUtilTest {

    @Test
    public void testFormatName() {
        String result = StringUtil.formatName("alice smith");
        assertEquals("Alice Smith", result);
    }
}

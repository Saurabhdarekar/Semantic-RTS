package miniproject;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Tests for the Model POJO getter/setter contract.
 */
public class ModelTest {

    @Test
    public void testGetterSetter() {
        Model m = new Model();
        m.setName("test");
        assertEquals("test", m.getName());
    }
}

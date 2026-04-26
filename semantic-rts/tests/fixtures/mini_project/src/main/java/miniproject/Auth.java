package miniproject;

public class Auth {

    public static boolean validatePassword(String password) {
        return password.length() >= 8;
    }

    public static boolean validateToken(String token) {
        return !token.startsWith("expired-");
    }
}

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;

class User {
    String userId;
    String name;

    boolean breakfast;
    boolean lunch;
    boolean snacks;
    boolean dinner;

    User(String userId, String name) {
        this.userId = userId;
        this.name = name;
    }
}

public class MessPunchSystem {

    // In-memory storage (Part-1)
    static HashMap<String, User> users = new HashMap<>();

    public static void main(String[] args) {

        // 🔹 Demo users (replace with DB later)
        users.put("231501047", new User("231501047", "GIRIDHAR U"));
        users.put("231501048", new User("231501048", "ARUN"));

        try {
            String userId = Files.readString(
                    Path.of("recognized_id.txt")
            ).trim();

            String token = punch(userId);

            // This output is captured by Python & shown in video
            System.out.println(token);

        } catch (IOException e) {
            System.out.println("ERROR: Face ID not received");
        }
    }

    // Punch logic (called by face recognition)
    static String punch(String userId) {

        if (!users.containsKey(userId)) {
            return "User Not Registered";
        }

        User user = users.get(userId);
        LocalDateTime now = LocalDateTime.now();
        String meal = getMealByTime(now.getHour());

        if (meal == null) {
            return "Mess Closed Now";
        }

        if (isDuplicate(user, meal)) {
            return "Duplicate Punch!";
        }

        setPunch(user, meal);
        return generateToken(user, now, meal);
    }

    // Identify meal by time
    static String getMealByTime(int hour) {
        if (hour >= 6 && hour < 10) return "BREAKFAST";
        if (hour >= 12 && hour < 15) return "LUNCH";
        if (hour >= 16 && hour < 18) return "SNACKS";
        if (hour >= 19 && hour < 22) return "DINNER";
        return null;
    }

    // Duplicate prevention
    static boolean isDuplicate(User user, String meal) {
        switch (meal) {
            case "BREAKFAST": return user.breakfast;
            case "LUNCH": return user.lunch;
            case "SNACKS": return user.snacks;
            case "DINNER": return user.dinner;
        }
        return false;
    }

    // Set punch flag
    static void setPunch(User user, String meal) {
        switch (meal) {
            case "BREAKFAST": user.breakfast = true; break;
            case "LUNCH": user.lunch = true; break;
            case "SNACKS": user.snacks = true; break;
            case "DINNER": user.dinner = true; break;
        }
    }

    // Token text (for video overlay)
    static String generateToken(User user, LocalDateTime time, String meal) {

        DateTimeFormatter formatter =
                DateTimeFormatter.ofPattern("dd-MM HH:mm:ss");

        return
                "Name: " + user.name +
                " | ID: " + user.userId +
                " | " + meal +
                " | " + time.format(formatter);
    }
}

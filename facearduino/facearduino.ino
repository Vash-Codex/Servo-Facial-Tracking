#include <Servo.h>

Servo myservo;

void setup() {
  Serial.begin(9600);
  myservo.attach(9);
  myservo.write(90);     // Start at center
  Serial.println("Send angle (0–180):");
}

void loop() {

  // Only read when a full line (string) is available  
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');   // read full line
    input.trim();                                  // remove spaces, \r, etc.

    // Ignore empty messages  
    if (input.length() == 0) return;

    // If not a pure number, ignore  
    if (!isNumber(input)) {
      Serial.print("Invalid input: ");
      Serial.println(input);
      return;
    }

    int angle = input.toInt();

    // Clamp 0–180
    angle = constrain(angle, 0, 180);

    myservo.write(angle);
    Serial.print("Servo moved to: ");
    Serial.println(angle);
  }
}

// Utility function to check if input contains only digits  
bool isNumber(String s) {
  for (unsigned int i = 0; i < s.length(); i++) {
    if (!isDigit(s[i])) return false;
  }
  return true;
}


Servo myservo;

void setup() {
  Serial.begin(9600);
  myservo.attach(9);
  myservo.write(90);
  Serial.println("Send angle (0–180):");
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.length() == 0) return;

    if (!isNumber(input)) {
      Serial.print("Invalid input: ");
      Serial.println(input);
      return;
    }

    int angle = input.toInt();
    angle = constrain(angle, 0, 180);

    myservo.write(angle);
    Serial.print("Servo moved to: ");
    Serial.println(angle);
  }
}

bool isNumber(String s) {
  for (unsigned int i = 0; i < s.length(); i++) {
    if (!isDigit(s[i])) return false;
  }
  return true;
}

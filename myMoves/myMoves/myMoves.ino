#include <Servo.h>

Servo servo1;
Servo servo2;
Servo hit;
int angle = 90;
const int servo1Pin = 9;
const int servo2Pin = 10;
const int hitpin = 11;
int correction = 0;
void BUB(int angle){
  servo1.write(30);
  delay(500);
  servo2.write(angle);
  delay(500);
  servo1.write(90);
  delay(500);
}
void hitman(){
  int speed = 50;
  int dd = 1310;
  // if(angle == 98) {speed = 45; dd =1170;}
  //     delay(100);
  //     hit.write(speed);
  //     delay(dd);
  //     hit.write(85);
  //     delay(500);
      // if(correction++ == 10){
      //   hit.write(95);
      //  delay(4500);
      //  hit.write(85);
      //  correction = 0;
      //  hit.write(0);
  delay(250);
  hit.write(85);
  delay(500);
  hit.write(90);
      
}
void setup() {
  Serial.begin(9600);

  servo1.attach(servo1Pin);
  servo2.attach(servo2Pin);
  hit.attach(hitpin);
  servo1.write(90);  
  servo2.write(90);   
  hit.write(90);
}
// # x -> hit once
// # y -> hit twice
// # z -> hit thrice
// # r -> rotate cw 90
// # i -> rotate ccw 90
// # p -> rotate cw 180
// # q -> rotate ccw 180
// # 1 -> rotate ccw 90 (blocked)
// # 2 -> rotate cw 90 (blocked)
// # 3 -> rotate ccw 180 (blocked)
// # 4 -> rotate cw 180 (blocked)

void executeR(){
  angle-=90;
  servo2.write(angle);
  delay(500);
}

void executeY(){

}

void executeI(){
  angle+=90;
  servo2.write(angle);
  delay(500);
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    switch(command){
      case 'r':
        executeR();
        break;

      case 'i':
        executeI();
        break;

      case 'p':
        angle-=180;
        servo2.write(angle);
        delay(500);
        break;
      
      case 'q':
        angle+=180;
        servo2.write(angle);
        delay(500);
        break;
      case '1':
      if(angle == 90){
        angle+=90;
        BUB(angle);
        angle-=108;
        servo2.write(angle);
        delay(500);
        angle+=35;
        BUB(angle);
        angle-=17;
        // servo2.write(angle);
        // delay(500);
        executeI();
      }else{
        angle+=108;
        BUB(angle);
        angle-=18;
        servo2.write(angle);
      }
        break;
      case '2':
      if(angle == 90){
        angle-=90;
        BUB(angle);
        angle+=108;
        servo2.write(angle);
        delay(500);
        angle-=30;
        BUB(angle);
        angle+=12;
        executeR();
      }else{
        angle -=108;
        BUB(angle);
        angle+=18;
        servo2.write(angle);
        delay(500);
      }
        break;
      
      case '3':
        angle-=180;
        BUB(angle);
        angle+=108;
        servo2.write(angle);
        delay(500);
        angle-=37;
        BUB(angle);
        angle+=17;
        executeR();
        break;
      
      case '4':
        angle+=180;
        BUB(angle);
        angle-=108;
        servo2.write(angle);
        delay(500);
        angle+=37;
        BUB(angle);
        angle-=17;
        executeI();
        break;

      case 'a':
        BUB(angle);
        break;
      case 'd':
        delay(1000);
        break;
      case 't':
        servo2.write(0);
        delay(500);
        break;
      case 's':
        hit.write(90);
        break;
      case 'c':
        hit.write(0);
        break;
      case 'v':
        hit.write(96);
        break;
    }

  //   // ----- Servo 1 Control -----
  //   if (command == '1') {
  //     servo1.write(90);   // Move to 90°
  //   }
  //   else if (command == '2') {
  //     servo1.write(30);   // Back to 30°
  //   }

  //   // ----- Servo 2 Control -----
  //   else if (command == '3') {
      // servo1.write(30);
      // delay(500);
      // servo2.write(180); 
      // delay(500);
      // servo1.write(90);
      // delay(500);
      // servo2.write(70);
      // delay(500);
      // servo1.write(30);
      // delay(500);
      // servo2.write(127);
      // delay(500);
      // servo1.write(90);
      // delay(500);
      // servo2.write(108);
  //   }
  //   else if (command == 'l' || command == 'L') {
  //     servo2.write(180);  // Move to 180°
  //     delay(500);
  //     // servo2.write(70);
  //     // delay(500);
  //     // servo2.write(127);
  //     // delay(500);
  //     // servo2.write(108);
  //   }
  //   else if (command == 'j' || command == 'J') {
  //     servo2.write(0);    // Move to 0°
  //   }
  //   else if (command == 'r' || command == 'r') {
  //     servo2.write(127);   // Move to 90°
  //     delay(500);
  //     servo2.write(108);
  //   }
  }
}
#include <Servo.h>

Servo blocker;
Servo rotor;
Servo hit;
int angle = 90;
const int blockerPin = 9;
const int rotorPin = 10;
const int hitpin = 11;
int correction = 0;

void BUB(int angle){
  blocker.write(30);
  delay(500);
  rotor.write(angle);
  delay(500);
  blocker.write(90);
  delay(500);
}

void hitman(){
  int speed = 0;
  int dd = 700;
  delay(100);
  hit.write(speed);
  delay(dd);
  hit.write(90);
  delay(475);
  if(correction++ ==2){
    hit.write(96);
    delay(4000);
    hit.write(70);
    delay(425);
    hit.write(90);
    correction = 0;
  }   
}

void setup() {
  Serial.begin(9600);

  blocker.attach(blockerPin);
  rotor.attach(rotorPin);
  hit.attach(hitpin);
  blocker.write(90);  
  rotor.write(90);   
  hit.write(96);
  delay(4000);
  hit.write(70);
  delay(425);
  hit.write(90);
}


void executeR(){
  angle-=90;
  rotor.write(angle);
  delay(500);
}


void executeI(){
  angle+=90;
  rotor.write(angle);
  delay(500);
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();

    int colon = line.indexOf(':');

    int id = line.substring(0, colon).toInt();
    char command = line.substring(colon + 1).charAt(0);
    switch(command){
      case 'r':
        executeR();
        break;

      case 'i':
        executeI();
        break;

      case 'p':
        angle-=180;
        rotor.write(angle);
        delay(500);
        break;
      
      case 'q':
        angle+=180;
        rotor.write(angle);
        delay(500);
        break;
      case '1':
      if(angle == 90){
        angle+=90;
        BUB(angle);
        angle-=108;
        rotor.write(angle);
        delay(500);
        angle+=40;
        BUB(angle);
        angle-=22;
        executeI();
      }else{
        angle+=115;
        BUB(angle);
        angle-=25;
        rotor.write(angle);
      }
        break;
      case '2':
      if(angle == 90){
        angle-=90;
        BUB(angle);
        angle+=108;
        rotor.write(angle);
        delay(500);
        angle-=22;
        BUB(angle);
        angle+=4;
        executeR();
      }else{
        angle -=100;
        BUB(angle);
        angle+=10;
        rotor.write(angle);
        delay(500);
      }
        break;
      
      case '3':
        angle-=180;
        BUB(angle);
        angle+=108;
        rotor.write(angle);
        delay(500);
        angle-=27;
        BUB(angle);
        angle+=7;
        executeR();
        break;
      
      case '4':
        angle+=180;
        BUB(angle);
        angle-=108;
        rotor.write(angle);
        delay(500);
        angle+=42;
        BUB(angle);
        angle-=22;
        executeI();
        break;
      case 'x':
        hitman();
        break;
      case 'y':
        hitman();
        hitman();
        break;
      case 'z':
        hitman();
        hitman();
        hitman();
        break;
      case 'a':
        BUB(angle);
        break;
      case 'd':
        delay(1000);
        break;
      case 't':
        rotor.write(0);
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
      case '0':
        blocker.write(90);  
        rotor.write(90);   
        hit.write(96);
        delay(4000);
        hit.write(70);
        delay(425);
        hit.write(90);
        correction = 0;
        angle = 90;
        break;
      case 'j':
        blocker.write(30);
    }
    Serial.print("DONE:");
    Serial.println(id);
  }
}

//iiripq342211
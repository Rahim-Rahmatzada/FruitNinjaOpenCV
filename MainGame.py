import cv2
import mediapipe as mp
import numpy as np
import random


class FruitNinjaGame:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        # Set larger resolution for capture
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.score = 0
        self.fruits = []
        _, self.frame = self.cap.read()
        self.height, self.width = self.frame.shape[:2]

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=2,
                                         min_detection_confidence=0.7,
                                         min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def create_fruit(self):
        x = random.randint(0, self.width - 1)
        y = self.height
        speed = random.randint(5, 15)
        return {'x': x, 'y': y, 'speed': speed}

    def update_fruits(self):
        for fruit in self.fruits:
            fruit['y'] -= fruit['speed']
        self.fruits = [fruit for fruit in self.fruits if fruit['y'] > 0]
        if random.random() < 0.05:
            self.fruits.append(self.create_fruit())

    def detect_hands(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        return results

    def check_collision(self, hand_landmarks, fruit):
        if hand_landmarks:
            for hand_landmark in hand_landmarks:
                for lm in hand_landmark.landmark:
                    lm_x, lm_y = int(lm.x * self.width), int(lm.y * self.height)
                    if (lm_x - fruit['x']) ** 2 + (lm_y - fruit['y']) ** 2 < 30 ** 2:  # Increased collision radius
                        return True
        return False

    def run(self):
        cv2.namedWindow('Fruit Ninja', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Fruit Ninja', 1280, 720)  # Set window size

        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                break

            self.frame = cv2.flip(self.frame, 1)
            hand_results = self.detect_hands(self.frame)

            self.update_fruits()

            fruits_to_remove = []
            for fruit in self.fruits:
                cv2.circle(self.frame, (int(fruit['x']), int(fruit['y'])), 20, (0, 255, 0), -1)
                if self.check_collision(hand_results.multi_hand_landmarks, fruit):
                    self.score += 1
                    fruits_to_remove.append(fruit)

            for fruit in fruits_to_remove:
                self.fruits.remove(fruit)

            # Draw hand landmarks
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        self.frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

            cv2.putText(self.frame, f"Score: {self.score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Fruit Ninja', self.frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    game = FruitNinjaGame()
    game.run()
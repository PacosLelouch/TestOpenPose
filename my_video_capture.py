
import queue, threading
import cv2

class MyVideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()
        
    def open(self, name):
        self.cap.open(name)
        
    def release(self):
        self.cap.release()
        
    def get(self, attr):
        return self.cap.get(attr)
        
    def isOpened(self):
        return self.cap.isOpened()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put((ret, frame))

    def read(self):
        return (False, None) if self.q.empty() else self.q.get()
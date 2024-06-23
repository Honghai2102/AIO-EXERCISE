class MyQueue:
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__queue = []

    def is_empty(self):
        return len(self.__queue) == 0

    def is_full(self):
        return len(self.__queue) >= self.__capacity

    def enqueue(self, value):
        if not self.is_full():
            self.__queue.append(value)
        else:
            print("Queue is full")

    def dequeue(self):
        if not self.is_empty():
            return self.__queue.pop(0)
        else:
            print("Queue is empty")

    def front(self):
        return self.__queue[0]


if __name__ == "__main__":
    # Test
    queue1 = MyQueue(capacity=5)
    queue1.enqueue(1)
    queue1.enqueue(2)
    print(queue1.is_full())
    print(queue1.front())
    print(queue1.dequeue())
    print(queue1.front())
    print(queue1.dequeue())
    print(queue1.is_empty())

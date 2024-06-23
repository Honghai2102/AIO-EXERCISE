class MyStack:
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__stack = []

    def is_empty(self):
        return len(self.__stack) == 0

    def is_full(self):
        return len(self.__stack) >= self.__capacity

    def push(self, value):
        if not self.is_full():
            self.__stack.append(value)
        else:
            print("Stack is full")

    def pop(self):
        if not self.is_empty():
            return self.__stack.pop()
        else:
            print("Stack is empty")

    def top(self):
        return self.__stack[-1]


if __name__ == "__main__":
    # Test
    stack1 = MyStack(capacity=5)
    stack1.push(1)
    stack1.push(2)
    print(stack1.is_full())
    print(stack1.top())
    print(stack1.pop())
    print(stack1.top())
    print(stack1.pop())
    print(stack1.is_empty())

class Ward:
    def __init__(self, name):
        self.__name = name
        self.__ward_list = []
        
    def add_person(self, job):
        self.__job = job
        self.__ward_list.append(self.__job)
        
    def count_doctor(self):
        count = 0
        for p in self.__ward_list:
            if isinstance(p, Doctor):
                count += 1
        return count
    
    def sort_age(self):
        self.__ward_list = sorted(self.__ward_list, key=lambda x: x._yob, reverse=True)
    
    def compute_average(self):
        count = 0
        total = 0
        for p in self.__ward_list:
            if isinstance(p, Teacher):
                count += 1
                total += p._yob
        return total / count
        
    def describe(self):
        print(f"Ward Name: {self.__name}")
        for p in self.__ward_list:
            p.describe()
        

class Person(Ward):
    def __init__(self, name, yob):
        self._name = name
        self._yob = yob
    

class Student(Person):
    def __init__(self, name, yob, grade):
        super().__init__(name, yob)
        self.__grade = grade
        
    def describe(self):
        print(f"Student - Name: {self._name} - YoB: {self._yob} - Grade: {self.__grade}")
        


class Teacher(Person):
    def __init__(self, name, yob, subject):
        super().__init__(name, yob)
        self.__subject = subject
    
    def describe(self):
        print(f"Teacher - Name: {self._name} - YoB: {self._yob} - Subject: {self.__subject}")
        


class Doctor(Person):
    def __init__(self, name, yob, specialist):
        super().__init__(name, yob)
        self.__specialist = specialist

    def describe(self):
        print(f"Doctor - Name: {self._name} - YoB: {self._yob} - Specialist: {self.__specialist}")
        
 

if __name__ == "__main__":
    print("Test case 2(a)")
    student1 = Student(name="studentA", yob=2010, grade="7")
    student1.describe()
    teacher1 = Teacher(name="teacherA", yob=1969, subject="Math")
    teacher1.describe()
    doctor1 = Doctor(name="doctorA", yob=1945, specialist="Endocrinologists")
    doctor1.describe()
    teacher2 = Teacher(name="teacherB", yob=1995, subject="History")
    teacher2.describe()
    doctor2 = Doctor(name="doctorB", yob=1975, specialist="Cardiologists")
    doctor2.describe()
    
    print("\nTest case 2(b)")
    ward1 = Ward(name="Ward1")
    ward1.add_person(student1)
    ward1.add_person(teacher1)
    ward1.add_person(teacher2)
    ward1.add_person(doctor1)
    ward1.add_person(doctor2)
    ward1.describe()
    
    print("\nTest case 2(c)")
    print(f"Number of doctors: {ward1.count_doctor()}")

    print("\nTest case 2(d)")
    print("After sorting Age of Ward1 people")
    ward1.sort_age()
    ward1.describe()
    
    print("\nTest case 2(e)")
    print (f"Average year of birth (teachers): {ward1.compute_average()}")

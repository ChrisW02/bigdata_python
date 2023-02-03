class Person:
    department = 'FinTech'

    def set_name(self,new_name):
        self.name = new_name

    def set_location(self,new_location):
        self.location = new_location


person = Person()
person.set_name('Chris Wang')
person.set_location('QD')

print('{} lives in {} and works in the department {}.'.format(person.name,person.location,person.department))

store1 = [7.50,20.00,12.50,6.00]
store2 = [9.30,14.75,10.00,9.50,12.00]
cheapset = map(min,store1,store2)
print(cheapset)
cheaplist = list(cheapset)
print(cheaplist)

def square(x):
    return x**2

squarelist = list(map(square,store1))
print(squarelist)

my_function = lambda a,b,c: a+b+c
print(my_function(1,2,3))

people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']
def split_title_and_name(person):
    return person.split()[0] + ' ' + person.split()[-1]

peoplelist1 = list(map(split_title_and_name, people))
peoplelist2 = list(map(lambda person: person.split()[0] + ' ' + person.split()[-1], people))
print(peoplelist2)
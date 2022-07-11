from django.db import models

class BusinessUnit(models.Model):
    name = models.CharField(max_length=30)

    def __str__(self):
        return self.name

class Line(models.Model):
    businessunit = models.ForeignKey(BusinessUnit, on_delete=models.CASCADE)
    name = models.CharField(max_length=30)

    def __str__(self):
        return self.name

class Category(models.Model):
    name = models.CharField(max_length=30)
    ename = models.CharField(max_length=30)

    def __str__(self):
        return self.name

class Station(models.Model):
    line = models.ForeignKey(Line, on_delete=models.CASCADE)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    number = models.CharField(max_length=30)
    name = models.CharField(max_length=50)
    ename = models.CharField(max_length=50)

    def __str__(self):
        return self.name

class MachineGroup(models.Model):
    name = models.CharField(max_length=30)

    def __str__(self):
        return self.name

class MachineType(models.Model):
    name = models.CharField(max_length=30)

    def __str__(self):
        return self.name      

class Issue(models.Model):

    TYPES = (
        ('ISSUE', '문제'),
        ('SOLUTION', '조치'),
    )

    type = models.CharField(max_length=30, choices=TYPES)
    businessunit = models.ForeignKey(BusinessUnit, on_delete=models.CASCADE)
    machinegroup = models.ForeignKey(MachineGroup, on_delete=models.CASCADE)
    machinetype = models.ForeignKey(MachineType, on_delete=models.CASCADE)
    content = models.CharField(max_length=100)
    econtent = models.CharField(max_length=100,blank=True)

    def __str__(self):
        return self.content
from django.contrib import admin

# Register your models here.

from .models import BusinessUnit, Line, Station, Category, MachineGroup, MachineType,Issue


admin.site.register(BusinessUnit)

@admin.register(Line)
class LineAdmin(admin.ModelAdmin):
    list_display = ('businessunit', 'name',)
    ordering = ('name',)
    search_fields = ('name',)

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ('name','ename',)
    ordering = ('name',)
    search_fields = ('name',)

@admin.register(Station)
class StationAdmin(admin.ModelAdmin):
    list_display = ('line','category', 'number', 'name', 'ename',)
    ordering = ('line','category', 'number',)
    search_fields = ('name',)

admin.site.register(MachineGroup)
admin.site.register(MachineType)

@admin.register(Issue)
class IssueAdmin(admin.ModelAdmin):
    list_display = ('type','businessunit', 'machinegroup',  'machinetype','content', 'econtent',)
    ordering = ('type','businessunit', 'machinegroup',  'machinetype','content',)
    search_fields = ('content',)










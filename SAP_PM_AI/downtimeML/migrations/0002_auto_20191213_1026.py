# Generated by Django 2.0.7 on 2019-12-13 01:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('downtimeML', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='issue',
            name='type',
            field=models.CharField(choices=[('ISSUE', '문제'), ('SOLUTION', '조치')], max_length=30),
        ),
    ]

from django.shortcuts import render
from django.http import HttpResponse


# Create your views here.

def service(request):
    return render(request, 'service.html')

def loan(request):
    return render(request, 'loan.html')

def house(request):
    return render(request, 'house.html')
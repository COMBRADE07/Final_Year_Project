from django.shortcuts import render


# Create your views here.

def algorithms(request):
    return render(request, 'algorithm.html')

from django.shortcuts import render

# Create your views here.
def datasets(request):
    return render(request, 'datasets.html')
from django.urls import path
from . import views

urlpatterns = [
    path('service/', views.service, name='service'),
    path('loan/', views.loan, name='loan'),
    path('house/', views.house, name='house'),

    ]
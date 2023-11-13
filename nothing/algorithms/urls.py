from django.urls import path
from . import views
urlpatterns = [
    path('algorithms/', views.algorithms, name='algorithms'),

    ]
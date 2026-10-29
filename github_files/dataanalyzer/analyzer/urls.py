from django.urls import path
from . import views
from .views import home, register, user_login, dashboard, user_logout


urlpatterns = [

    path('', home, name='home'),

    path('register/', register, name='register'),

    path('login/', user_login, name='login'),

    path('dashboard/', dashboard, name='dashboard'),

    path('logout/', user_logout, name='logout'),

    path('upload/', views.upload_dataset, name='upload_dataset'),

    path(
    'analyze/<int:dataset_id>/',
    views.analyze_dataset,
    name='analyze_dataset'
),
    path(
    'charts/<int:dataset_id>/',
    views.generate_charts,
    name='generate_charts'
),

    path(
    'ai-insights/<int:dataset_id>/',
    views.ai_insights,
    name='ai_insights'
),

   path(
    'ask-ai/<int:dataset_id>/',
    views.ask_ai,
    name='ask_ai'
),

path(
    'ai-report/<int:dataset_id>/',
    views.ai_report,
    name='ai_report'
),

path('delete/<int:dataset_id>/', views.delete_dataset, name='delete_dataset'),
path('download/<int:dataset_id>/', views.download_dataset, name='download_dataset'),
path('rename/<int:dataset_id>/', views.rename_dataset, name='rename_dataset'),
path('details/<int:dataset_id>/', views.dataset_details, name='dataset_details'),

    path('manifest.json', views.manifest, name='manifest'),
    path('sw.js', views.service_worker, name='service_worker'),
    path('favicon.ico', views.favicon, name='favicon'),
]

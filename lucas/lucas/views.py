from django.shortcuts import render

def intro(request):
    return render(request, "lucas/intro.html")

def index(request):
    return render(request, 'lucas/index.html')

def about(request):
    return render(request, 'lucas/about.html')

def service(request):
    return render(request, 'lucas/service.html')

def contact(request):
    return render(request, 'lucas/contact.html')

def feature(request):
    return render(request, 'lucas/feature.html')

def quote(request):
    return render(request, 'lucas/quote.html')

def team(request):
    return render(request, 'lucas/team.html')

def nopage(request):
    return render(request, 'lucas/404.html')
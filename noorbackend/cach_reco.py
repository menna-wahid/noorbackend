from django.http import JsonResponse


def cach(request):

    return JsonResponse(data= {'msg': 'hi'})
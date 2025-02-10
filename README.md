IA - MIST

Étapes pour démarrer

1. Construire l'image Docker

Exécutez la commande suivante pour construire l'image Docker :
docker build -t ia-mist:latest .

2. Lancer le conteneur Docker

Exécutez la commande suivante pour démarrer le conteneur :
docker run -d --name ia-mist --rm -p 8000:8000 ia-mist:latest

Le conteneur sera exécuté en arrière-plan.

L'API sera accessible sur : http://localhost:8000.

Pour arrêter le conteneur, exécutez simplement la commande suivante 
docker stop ia-mist


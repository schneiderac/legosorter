# legosorter

trigger.py ist der Teil des Programmes, der immer läuft und schaut, ob in einem bestimmten Verzeichnis das neueste Bild (irgendeine Datei) angelegt wird. Das Modul das dafür verwendet wird heißt watchdog. Von hier aus wird der imageclassfier aufgerufen und auch die Verbindung zum Raspi als Server herstellt.

imageclassifier.py ist eine Klasse, die das vortrainierte tensorflow-for-poets-2 aufrufen, mit einem Bild füttern und die Ergebnisse der Klassifikation zurückliefern kann.

picture-taking.py ist für das Bild machen zuständig. Nötig für die Bewegungserkennung ist opencv, was ein bisschen kompliziert zu installieren ist, aber definitiv einen Blick wert, wenn es um Objekterkennung geht.


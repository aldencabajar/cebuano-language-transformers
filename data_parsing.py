import subprocess
import xml.sax
import re
import os
import time
# change this to reflect the xml file path 
path_to_file = '/mnt/d/cebwiki-latest-pages-articles-multistream.xml.bz2'
lines_list = []

class WikiXmlHandler(xml.sax.handler.ContentHandler):
    """Content handler for Wiki XML data using SAX"""
    def __init__(self):
        xml.sax.handler.ContentHandler.__init__(self)
        self._buffer = None
        self._values = {}
        self._current_tag = None
        self._pages = []
        self._true_doc_count = 0

    def characters(self, content):
        """Characters between opening and closing tags"""
        if self._current_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        """Opening tag of element"""
        if name in ('title', 'text'):
            self._current_tag = name
            self._buffer = []

    def endElement(self, name):
        """Closing tag of element"""
        if name == self._current_tag:
            self._values[name] = ' '.join(self._buffer)

        # print(re.search(self._values['text'], '(?i)Lsjbot(?-i)'))

        if name == 'page':
            # add a special filter to remove articles that were made by bots
            if not re.search('Lsjbot', self._values['text']): 
                self._true_doc_count += 1
                self._pages.append((self._values['title'], self._values['text']))

if __name__ == '__main__': 
    handler = WikiXmlHandler()
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)
    start = time.time()
    for line in subprocess.Popen(['bzcat'], 
                                stdin = open(path_to_file), 
                                stdout = subprocess.PIPE).stdout:
        parser.feed(line)
        if len(handler._pages) > 1000:
            break

    end = time.time()
            
    # check parsed documents
    with open('parsed_wiki.txt', 'w') as f:
        for i, item in enumerate(handler._pages):
            f.write("%s\n" % '{}===Page {}==========================\n{}'.format(item[0], i, item[1]))
    
    print("A total of", handler._true_doc_count, "non-bot written pages were retrieved.")
    total_time = end - start 
    print("processed in", total_time, "seconds")

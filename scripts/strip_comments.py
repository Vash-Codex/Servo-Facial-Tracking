import ast 
import io 
import os 
import re 
import sys 
import tokenize 

ROOT =os .path .abspath (os .path .join (os .path .dirname (__file__ ),'..'))
SKIP_DIRS ={'.venv','build','__pycache__','.git','.github'}
TARGET_EXTS ={'.py','.ino','.html','.htm','.gitignore'}


def remove_py_docstrings_and_comments (source :str )->str :
    try :
        mod =ast .parse (source )
    except Exception :
        return source 

    doc_lines =set ()
    for node in ast .walk (mod ):
        if isinstance (node ,(ast .Module ,ast .ClassDef ,ast .FunctionDef ,ast .AsyncFunctionDef )):
            if getattr (node ,'body',None ):
                first =node .body [0 ]
                if isinstance (first ,ast .Expr )and isinstance (getattr (first ,'value',None ),ast .Constant )and isinstance (first .value .value ,str ):
                    start =getattr (first ,'lineno',None )
                    end =getattr (first ,'end_lineno',start )
                    if start is not None :
                        for ln in range (start ,end +1 ):
                            doc_lines .add (ln )

    out_tokens =[]
    sio =io .StringIO (source )
    try :
        gen =tokenize .generate_tokens (sio .readline )
    except Exception :
        return source 

    for toknum ,tokval ,start ,end ,line in gen :
        if toknum ==tokenize .COMMENT :
            continue 
        if toknum ==tokenize .STRING :

            if start [0 ]in doc_lines :
                continue 
        out_tokens .append ((toknum ,tokval ))

    try :
        return tokenize .untokenize (out_tokens )
    except Exception :
        return source 


def remove_c_style_comments (text :str )->str :

    text =re .sub (r'/\*[\s\S]*?\*/','',text )

    text =re .sub (r'//.*$','',text ,flags =re .MULTILINE )
    return text 


def remove_html_comments (text :str )->str :
    return re .sub (r'<!--([\s\S]*?)-->','',text )


def remove_hash_comments_in_file (text :str )->str :

    return re .sub (r'^[ \t]*#.*$\n?','',text ,flags =re .MULTILINE )


def process_file (path :str )->bool :
    rel =os .path .relpath (path ,ROOT )
    _ ,ext =os .path .splitext (path )
    with open (path ,'r',encoding ='utf-8',errors ='surrogateescape')as f :
        src =f .read ()

    orig =src 
    if ext =='.py':
        src =remove_py_docstrings_and_comments (src )

        src =remove_hash_comments_in_file (src )
    elif ext in ('.html','.htm'):
        src =remove_html_comments (src )
    elif ext =='.ino':
        src =remove_c_style_comments (src )

        src =remove_hash_comments_in_file (src )
    elif os .path .basename (path )=='.gitignore':
        src =remove_hash_comments_in_file (src )

    if src !=orig :
        with open (path ,'w',encoding ='utf-8',errors ='surrogateescape')as f :
            f .write (src )
        print (f"Cleaned: {rel}")
        return True 
    return False 


if __name__ =='__main__':
    changed =0 
    for root ,dirs ,files in os .walk (ROOT ):
        parts =set (root .split (os .sep ))
        if parts &SKIP_DIRS :
            continue 
        for fn in files :
            if fn .startswith ('.')and fn not in ('.gitignore',):
                continue 
            path =os .path .join (root ,fn )
            _ ,ext =os .path .splitext (fn )
            if ext .lower ()in TARGET_EXTS or fn =='.gitignore':
                try :
                    print (f"Processing: {os.path.relpath(path, ROOT)}")
                    if process_file (path ):
                        changed +=1 
                except Exception as e :
                    print (f"Error processing {path}: {e}",file =sys .stderr )
    print (f"Done. Files changed: {changed}")

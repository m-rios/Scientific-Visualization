//
// Created by arjun on 22-2-18.
//

#ifndef SCIENTIFIC_VISUALIZATION_UI_H
#define SCIENTIFIC_VISUALIZATION_UI_H



 #ifdef __cplusplus

 #include <string>

 class Foo
 {
 public:
    Foo( int32_t a );
    ~Foo();
    std::string toString();
 private:
    int32_t k;
 };

 #endif

 #ifdef __cplusplus
 extern "C"
 {
 #endif
void*  getFoo( int32_t a );
 void   destroyFoo( void *foo );
 void   printString( void *foo );

#ifdef __cplusplus
 }
 #endif
#endif //SCIENTIFIC_VISUALIZATION_UI_H

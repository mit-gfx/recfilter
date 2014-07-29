#include "recfilter.h"

using std::string;
using std::vector;
using std::cerr;
using std::endl;
using std::ostream;
using std::runtime_error;

using namespace Halide;

Arguments::Arguments(string app_name, int argc, char** argv) :
    width  (256),
    block  (32),
    iterations(1),
    nocheck(false)
{
    string desc = "\nUsage\n ./"+ app_name;
    desc.append(string(
                "[-width|-w w] [-tile|-block|-b|-t b] [-iter i] [-nocheck] [-help]\n\n"
                "\twidth\t  width of input image [default = 256]\n"
                "\ttile\t   tile width for splitting each dimension image [default = 32]\n"
                "\tnocheck\t do not check against reference solution [default = false]\n"
                "\titer\t  number of profiling iterations [default = 1]\n"
                "\thelp\t  show help message\n"
                )
            );

    try {
        for (int i=1; i<argc; i++) {
            string option = argv[i];

            if (!option.compare("-help") || !option.compare("--help")) {
                throw runtime_error("Showing help message");
            }

            else if (!option.compare("-nocheck") || !option.compare("--nocheck")) {
                nocheck = true;
            }

            else if (!option.compare("-iter") || !option.compare("--iter")) {
                if ((i+1) < argc)
                    iterations = atoi(argv[++i]);
                else
                    throw runtime_error("-iterations requires a int value");
            }

            else if (!option.compare("-w") || !option.compare("--w") || !option.compare("-width") || !option.compare("--width")) {
                if ((i+1) < argc) {
                    width = atoi(argv[++i]);
                }
                else
                    throw runtime_error("-width requires an integer value");
            }

            else if (!option.compare("-b") || !option.compare("--b") || !option.compare("-block") || !option.compare("--block") ||
                     !option.compare("-t") || !option.compare("--t") || !option.compare("-tile") || !option.compare("--tile"))
            {
                if ((i+1) < argc)
                    block = atoi(argv[++i]);
                else
                    throw runtime_error("-block requires an integer value");
            }

            else {
                throw runtime_error("Bad command line option " + option);
            }
        }

        if (width%block)
            throw runtime_error("Width should be a multiple of block size");

    } catch (runtime_error& e) {
        cerr << endl << e.what() << endl << desc << endl;
        exit(EXIT_FAILURE);
    }
}

// -----------------------------------------------------------------------------

ostream &operator<<(ostream &s, CheckResult v) {
    Image<float> ref = v.ref;
    Image<float> out = v.out;

    assert(ref.width() == out.width());
    assert(ref.height() == out.height());
    assert(ref.channels() == out.channels());

    int width = ref.width();
    int height = ref.height();
    int channels = ref.channels();

    Image<float> diff(width, height, channels);

    float diff_sum = 0.0f;
    float max_val  = 0.0f;

    for (int z=0; z<channels; z++) {
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                diff(x,y,z) = float(ref(x,y,z)) - float(out(x,y,z));
                diff_sum += abs(diff(x,y,z) * diff(x,y,z));
                max_val = std::max(ref(x,y,z), max_val);
            }
        }
    }
}

ostream &operator<<(ostream &s, CheckResultVerbose v) {
    Image<float> ref = v.ref;
    Image<float> out = v.out;

    assert(ref.width() == out.width());
    assert(ref.height() == out.height());
    assert(ref.channels() == out.channels());

    int width = ref.width();
    int height = ref.height();
    int channels = ref.channels();

    Image<float> diff(width, height, channels);

    float diff_sum = 0.0f;
    float max_val  = 0.0f;

    for (int z=0; z<channels; z++) {
        for (int y=0; y<height; y++) {
            for (int x=0; x<width; x++) {
                diff(x,y,z) = float(ref(x,y,z)) - float(out(x,y,z));
                diff_sum += abs(diff(x,y,z) * diff(x,y,z));
                max_val = std::max(ref(x,y,z), max_val);
            }
        }
    }
    float mse  = diff_sum/float(width*height*channels);

    s << "Reference" << "\n" << ref << "\n";
    s << "Halide output" << "\n" << out << "\n";
    s << "Difference " << "\n" << diff << "\n";
    s << "Mean sq error = " << mse << "\n\n";

    return s;
}

ostream &operator<<(ostream &s, RecFilter r) {
    map<string,Func> funcs = r.funcs();
    map<string,Func>::iterator f  = funcs.begin();
    map<string,Func>::iterator fe = funcs.end();
    while (f != fe) {
        s << f->second << endl;
        f++;
    }
    return s;
}

ostream &operator<<(ostream &s, Func f) {
    s << f.function();
    return s;
}

ostream &operator<<(ostream &s, Internal::Function f) {
    if (f.has_pure_definition()) {
        vector<Expr> pure_value = f.values();
        s << "Func " << f.name() << ";\n";
        for (int v=0; v<pure_value.size(); v++) {
            vector<string> args = f.args();
            s << f.name() << "(";
            for (int i=0; i<args.size(); i++) {
                s << args[i];
                if (i<args.size()-1)
                    s << ",";
            }
            if (pure_value.size()>1)
                s << ")[" << v << "]";
            else
                s << ")";
            s << " = " << pure_value[v] << "\n";
        }

        // reduction definitions
        for (int j=0; j<f.reductions().size(); j++) {
            vector<Expr> reduction_value = f.reductions()[j].values;
            for (int v=0; v<reduction_value.size(); v++) {
                vector<Expr> args = f.reductions()[j].args;
                s << f.name() << "(";
                for (int i=0; i<args.size(); i++) {
                    s << args[i];
                    if (i<args.size()-1)
                        s << ",";
                }
                if (reduction_value.size()>1)
                    s << ")[" << v << "]";
                else
                    s << ")";
                s << " = " << reduction_value[v];
                if (f.reductions()[j].domain.defined()) {
                    s << " with  ";
                    for (int k=0; k<f.reductions()[j].domain.domain().size(); k++) {
                        string r = f.reductions()[j].domain.domain()[k].var;
                        s << r << "("
                            << f.reductions()[j].domain.domain()[k].min   << ","
                            << f.reductions()[j].domain.domain()[k].extent<< ") ";
                    }
                }
                s << "\n";
            }
        }
    }
    return s;
}

// -----------------------------------------------------------------------------


Timer::Timer(string name) {
    m_Name = name;
    start();
}

Timer::~Timer(void) {
    stop();
}

void Timer::start(void) {
    m_TmStart = milliseconds();
}

Timer::t_time Timer::stop(void) {
    Timer::t_time tm = elapsed();
    Timer::t_time h  = ( tm/(1000*60*60));
    Timer::t_time m  = ((tm/(1000*60)) % 60);
    Timer::t_time s  = ((tm/1000)      % 60);
    Timer::t_time ms = tm % 1000;
    cerr << m_Name.c_str() << ": " << h << "h " << m
        << "m " << s << "s " << ms << "ms" << endl;
    return tm;
}

Timer::t_time Timer::elapsed(void) {
    return (milliseconds() - m_TmStart);
}

Timer::t_time Timer::milliseconds(void) {
    static bool init = false;
#ifdef WIN32
    static Timer::t_time freq;
    if (!init) {
        init = true;
        LARGE_INTEGER lfreq;
        assert(QueryPerformanceFrequency(&lfreq) != 0);
        freq = Timer::t_time(lfreq.QuadPart);
    }
    LARGE_INTEGER tps;
    QueryPerformanceCounter(&tps);
    return (Timer::t_time(tps.QuadPart)*1000/freq);
#else
    struct timeval        now;
    static struct timeval start;
    if (!init) {
        gettimeofday(&start, NULL);
        init = true;
    }
    gettimeofday(&now, NULL);
    uint ms = uint((now.tv_sec-start.tv_sec)*1000+(now.tv_usec-start.tv_usec)/1000);
    return ms;
#endif
}
